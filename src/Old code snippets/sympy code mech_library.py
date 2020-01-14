        t = sym.symbols('t', real=True)
        du_1 = torch.tensor([])
        Expression = library_config['input_expr'] 
        for order in range(max_order+1):
            if order > 0:
                Expression = Expression.diff(t)
            
            x = vedg.Eval_Array_From_Expression(data.detach(), t, Expression)
            du_1 = torch.cat((du_1, x), dim=1)
            
        library_config['input_theta'] = du_1

....

    du_2 = prediction #.clone()
    for order in range(1, max_order+1):
        y = grad(du_2[:, [order-1]], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        #removed '[:, 1:2]' from very end of grad()[] statement
        du_2 = torch.cat((du_2, y), dim=1)