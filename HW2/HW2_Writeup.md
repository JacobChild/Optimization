# HW 2: Bracketing and Pinpoint Algorithms

### 1.3

See Attached Code



### 1.4

- **Slanted Quadratic Function:** 
  
  - Optimal Step Length: 4.0
  
  - Point: (-2,2)
  
  - Function Calls: 6

- **Rosenbrock Function:**
  
  - Optimal Step Length: approximately 0; 1.57e-30
  
  - Point: (apx 0, 2)
  
  - Function Calls: 209

- **Jones Function:**
  
  - Optimal Step Length: 0.59375
  
  - Point: (1.59375, 2.1875)
  
  - Function Calls: 24 
  
  

**What I learned:** It was pretty difficult to figure out initially, especially pinpointing as I had a bit of a harder time figuring out what it was supposed to be doing. Switching from attempting the cubic fit to a simple bisection method helped things work smoother, although I still had some errors jumping wrong directions etc. The bisection method will cause more function calls as it isn't fitting a line or using the derivatives, so I will need to fix that. The Rosenbrock function seemed to start at the minimum, and then my first step jumped away and it was really difficult to work its way back. This shows me that direction is really important as guessing the wrong way can add significant difficulty. 
