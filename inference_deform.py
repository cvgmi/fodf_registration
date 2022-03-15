import torch
from layers.resample import resample_fa
import nibabel as nib
from layers.resample import SpatialTransformer
from layers.reorient import DeformationJacobian

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser("inference")
    parser.add_argument("model", type=str)
    parser.add_argument("moving", type=str)
    parser.add_argument("fixed", type=str)
    parser.add_argument("apply_to", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("--fa", action='store_true')

    args = parser.parse_args()

    from dataset import load_subject

    moving = load_subject(args.moving)
    fixed = load_subject(args.fixed)
    apply_to = load_subject(args.apply_to)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moving = moving.to(device).unsqueeze(0).unsqueeze(1)
    fixed = fixed.to(device).unsqueeze(0).unsqueeze(1)
    apply_to = apply_to.to(device).unsqueeze(0).unsqueeze(1)

    jacobian = DeformationJacobian(moving.shape[2:-1]).to(device)
    with torch.no_grad():
        from model import MVCRegNet
        model = MVCRegNet()
        model.load_state_dict(torch.load(args.model))
        model.eval()
        model = model.to('cuda')

        combined = torch.cat([moving, fixed], axis=1)
        fodf, d = model(combined)
        J = torch.det(jacobian(d))
        print( torch.min(J) )

    transformer = SpatialTransformer(fodf.shape[1:-1]).to(device)
    warped = transformer(apply_to, d)
    header_img = nib.load(args.fixed)
    nib.save(nib.Nifti1Image(warped[0][0].cpu().numpy(), header_img.affine), args.output)
