import numpy as np
import visvalingamwyatt as vw

class GenPivots:
    def __init__(self, max_pts=[10, 2, 30], map_region=(30, -30, 15, -15), vm_thre=2.0, resolution=0.15):
        self.max_pts = max_pts
        self.map_region = map_region
        self.vm_thre = vm_thre
        self.resolution = resolution
        
    def pivots_generate(self, map_vectors):
        pivots_single_frame =  {0:[], 1:[], 2:[]}
        lengths_single_frame =  {0:[], 1:[], 2:[]}
        for ii, vec in enumerate(map_vectors):
            pts = np.array(vec["pts"]) * self.resolution  # 转成 m
            pts = pts[:, ::-1]
            cls = vec["type"]
        
            # If the difference in x is obvious (greater than 1m), then rank according to x. 
            # If the difference in x is not obvious, rank according to y.
            if (np.abs(pts[0][0]-pts[-1][0])>1 and pts[0][0]<pts[-1][0]) \
                or (np.abs(pts[0][0]-pts[-1][0])<=1 and pts[0][1]<pts[-1][1]): 
                pts = pts[::-1]
        
            simplifier = vw.Simplifier(pts)
            sim_pts = simplifier.simplify(threshold=self.vm_thre)
            length = min(self.max_pts[cls], len(sim_pts))
            padded_pts = self.pad_pts(sim_pts, self.max_pts[cls])
            pivots_single_frame[cls].append(padded_pts)
            lengths_single_frame[cls].append(length)

        for cls in [0, 1, 2]:
            new_pts = np.array(pivots_single_frame[cls])
            if new_pts.size > 0:
                new_pts[:, :, 0] = new_pts[:, :, 0] / (2 * self.map_region[0])  # normalize
                new_pts[:, :, 1] = new_pts[:, :, 1] / (2 * self.map_region[2])
            pivots_single_frame[cls] = new_pts
            lengths_single_frame[cls] = np.array(lengths_single_frame[cls])
            
        return pivots_single_frame, lengths_single_frame
    
    def pad_pts(self, pts, tgt_length):
        if len(pts) >= tgt_length:
            return pts[:tgt_length]
        pts = np.concatenate([pts, np.zeros((tgt_length-len(pts), 2))], axis=0)
        return pts
