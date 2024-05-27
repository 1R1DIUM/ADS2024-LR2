from PIL import Image
import numpy as np
import pickle


class workImage:
    def __init__(self,input_filename_format) -> None:
        img = Image.open(input_filename_format)
        self.rgb_img_arr = np.array(img,np.uint8)
        self.ycbcr_img_arr = None
        self.img_size = img.size
        
        
    def rgb_to_YCbCr(self,save_in_class = True) ->np.ndarray:
        '''Функция для JPEG - конвертации RGB изображения в виде массива (N,M,3) в классе 
        в YCbCr массив (N,M,3)
        
        save_in_class – необходимо ли записывать в класс полученную матрицу
        '''
        zeros_arr  = np.zeros_like(self.rgb_img_arr,np.int16) #!Пустой массив нулей

        for line in range(len(zeros_arr)):
            for pixel in range(len(zeros_arr[0])):  
            
                R = self.rgb_img_arr[line][pixel][0]
                G = self.rgb_img_arr[line][pixel][1]
                B = self.rgb_img_arr[line][pixel][2]
                #!преобразование значений
                # zeros_arr[line][pixel][0]  = min(max(0,round(0.299*R + 0.587*G + 0.0114 * B)) ,255)
                # zeros_arr[line][pixel][1] = min(max(0,round(-0.1687*R - 0.3313 *G + 0.5 * B + 128)),255)
                # zeros_arr[line][pixel][2] = min(max(0,round(0.5*R - 0.4187* G - 0.0813*B+ 128)),255)
                
                zeros_arr[line][pixel][0]  = 0.299*R + 0.587*G + 0.0114 * B
                zeros_arr[line][pixel][1] = -0.1687*R - 0.3313 *G + 0.5 * B + 128
                zeros_arr[line][pixel][2] = 0.5*R - 0.4187* G - 0.0813*B+ 128
                
        zeros_arr[zeros_arr > 255] = 255
        zeros_arr[zeros_arr < 0] = 0
        ycbcr_arr = np.array(zeros_arr,dtype=np.uint8)
        
           
        if save_in_class:
            self.ycbcr_img_arr = ycbcr_arr

        return ycbcr_arr

    @staticmethod
    def YCbCr_to_rgb(ycbcr_array) ->np.ndarray:
        '''Функция для JPEG - конвертации RGB изображения в виде массива RGB (N,M,3) в классе 
        в YCbCr массив (N,M,3) '''
        
        rgb_arr = np.zeros_like(ycbcr_array,np.uint8)
        
        #? Преобразование
        for line in range(len(rgb_arr)):
            for pixel in range(len(rgb_arr[0])):
                Y = ycbcr_array[line][pixel][0]
                Cb = ycbcr_array[line][pixel][1]
                Cr = ycbcr_array[line][pixel][2]
                   
                rgb_arr[line][pixel][0] = min(max(0,round(Y + 1.402*(Cr-128))),255)
                rgb_arr[line][pixel][1] = min(max(0,round(Y - 0.344*(Cb-128) - 0.7714*(Cr-128))),255)
                rgb_arr[line][pixel][2] = min(max(0,round(Y + 1.772 *(Cb-128))),255)
        return rgb_arr

    @staticmethod
    def zig_zag_scanning(input_matrix : np.ndarray):
        '''Функция, которая принимает матрицу N*M и вовзращает одномерный массив длиной N*M '''
        rows,collums = len(input_matrix),len(input_matrix[0])
        num = rows*collums
        
        one_d_array = []
        i,j = 0,0
        
        while i < rows and j < collums and len(one_d_array) < num:
            one_d_array.append(input_matrix[i][j])
            if (i+j) % 2 ==0 : #!Если сумма четная
                if (i-1) in range(rows) and (j+1) not in range(collums): 
                    i = i +1                                             #^ Двигаемся вниз
                elif (i-1) not in range(rows) and (j+1) in range(collums): 
                    j = j + 1                                             #^ Двигаемся вправо        
                elif (i-1) not in range(rows) and (j+1) not in range(collums):
                    i = i +1                                             #^ Двигаемся вниз
                else:
                    #! Движение по диагонали вправо вверх
                    i = i - 1
                    j = j + 1
            else: #!Если сумма нечетная
                if (i+1) in range(rows) and (j-1) not in range(collums):
                    i = i + 1                                             #^ Двигаемся вниз
                elif (i+1) not in range(rows) and (j-1) in range(collums):
                    j = j + 1                                             #^ Двигаемся вправо
                elif (i+1) not in range(rows) and (j-1) not in range(collums):
                    j = j + 1                                             #^ Двигаемся вправо
                else:
                    #! Движение по диагонали слево вниз
                    i = i + 1
                    j = j - 1

        return np.array(one_d_array)

    
    @staticmethod
    def zig_zag_writiing(input_sequence,x_size,y_size):
        '''
        Функция для записи матрицы из последовательности, полученной обходом зигзагом
        '''
        rows = y_size
        collums = x_size
        num = x_size*y_size
        out_matrix = np.zeros((x_size,y_size))
        
        i,j = 0,0
        num_written = 0
        while i < rows and j < collums and num_written < num:
            out_matrix[i][j] = input_sequence[num_written]
            num_written+=1
            
            if (i+j) % 2 ==0 : #!Если сумма четная
                if (i-1) in range(rows) and (j+1) not in range(collums): 
                    i = i +1                                             #^ Двигаемся вниз
                elif (i-1) not in range(rows) and (j+1) in range(collums): 
                    j = j + 1                                             #^ Двигаемся вправо        
                elif (i-1) not in range(rows) and (j+1) not in range(collums):
                    i = i +1                                             #^ Двигаемся вниз
                else:
                    #! Движение по диагонали вправо вверх
                    i = i - 1
                    j = j + 1
            else: #!Если сумма нечетная
                if (i+1) in range(rows) and (j-1) not in range(collums):
                    i = i + 1                                             #^ Двигаемся вниз
                elif (i+1) not in range(rows) and (j-1) in range(collums):
                    j = j + 1                                             #^ Двигаемся вправо
                elif (i+1) not in range(rows) and (j-1) not in range(collums):
                    j = j + 1                                             #^ Двигаемся вправо
                else:
                    #! Движение по диагонали слево вниз
                    i = i + 1
                    j = j - 1
        return out_matrix
            
    @staticmethod
    def downSampling_channel(channel_matrix : np.ndarray, Cx:int, Cy:int, method = 'deletion') -> np.ndarray:
        '''Функция даунсэмплинга канала изображения
        Входные данные: channed_matrix - матрица канала изображения [N,M], Cx Cy - коэф. сжатия по x и по y соотвественно
        
            method = deletion - удаление строк и столбов 
        
            method = average - замена на пиксель со средним значением цвета блока
        
            method = close - замена на пиксель с цветом, ближайши к среднему значениею блока
        
        Выходные данные: матрица канала размером [N/Cx,M/Cy]
        '''
        X = len(channel_matrix[0])
        Y = len(channel_matrix)
        
        
        newX = int(np.floor(X/Cx))
        newY = int(np.floor(Y/Cy))
        new_ch_matrix = np.zeros((newY,newX),dtype=channel_matrix.dtype)

        new_ch_x = 0 #! Индекс для новой матрицы
        new_ch_y = 0   
        if method == 'deletion':
            
            
            for x in range(0,X-Cx+1,Cx):
                for y in range(0,Y-Cy+1,Cy):
                    new_ch_matrix[new_ch_y][new_ch_x] = channel_matrix[y][x]
                    new_ch_y +=1
                new_ch_x += 1
                new_ch_y = 0 
            return new_ch_matrix
        
        if method == 'average':
            for x in range(0,X-Cx+1,Cx):
                for y in range(0,Y-Cy+1,Cy):
                    pixel_block = channel_matrix[y:y+Cy,x:x+Cx] #! получаем блок пикселей 
                    color = np.sum(pixel_block) / len(pixel_block[0]) / len(pixel_block) #!вычисляем среднее значение цвета блока пикселей
                    new_ch_matrix[new_ch_y][new_ch_x] = color 
                    new_ch_y+=1
                new_ch_y= 0
                new_ch_x +=1
            return new_ch_matrix
        
        if method == 'close':
            for x in range(0,X-Cx+1,Cx):
                for y in range(0,Y-Cy+1,Cy):
                    pixel_block = np.array(channel_matrix[y:y+Cy,x:x+Cx].copy(), np.uint8) #!Копируем блок
                    medium = np.sum(pixel_block) / len(pixel_block[0]) / len(pixel_block) #!Вычисляем среднее
                    diff_block= np.array(pixel_block - np.uint8(medium),np.int16) #! Вычисляем матрицу разницы
                    color_ind = np.absolute(diff_block.flatten()).argmin() #!вычисляем индекс из списка абсютных разностей
                    color =  pixel_block.flatten()[color_ind] #! определяем цвет по индексу
                    new_ch_matrix[new_ch_y][new_ch_x] = color
                    new_ch_y +=1
                new_ch_y = 0
                new_ch_x +=1 
            return new_ch_matrix
    
    
    def upSampling(image_matrix : np.ndarray, Cx:int,Cy:int) -> np.ndarray:
        '''Функция, которая увеличивает размер исходного изображения [N,M] до размера 
        [N*Cx,M*Cy]. Каждый пиксель заменяется блоком пикселей с тем же значением цвета
        
        Входные данные: image_matrix — матрица формы (N,M,3), Cx Cy - коэф. расширения по осям x и y соответсвнно
        
        Выходные данные: матрица формы (N*Cx,M*Cy,3)
        
        '''
        X = len(image_matrix[0])
        Y = len(image_matrix)
        
        newX = X *Cx
        newY = Y *Cy
        new_up_matrix = np.zeros((newY,newX,3),dtype=image_matrix.dtype)
        
        for x in range(newX):
            for y in range(newY):
                new_up_matrix[y][x] = image_matrix[int(y/Cy)][int(x/Cx)]
                
        return new_up_matrix
    def upSampling_channel(channel_matrix : np.ndarray, Cx:int,Cy:int) -> np.ndarray:
        X = len(channel_matrix[0])
        Y = len(channel_matrix)
        
        newX = X *Cx
        newY = Y *Cy
        new_up_matrix = np.zeros((newY,newX),dtype=channel_matrix.dtype)
        
        for x in range(newX):
            for y in range(newY):
                new_up_matrix[y][x] = channel_matrix[int(y/Cy)][int(x/Cx)]
                
        return new_up_matrix

class JPEG:
    @staticmethod
    def split_to_8_8_blocks(color_matrix:np.ndarray):
        #color_matrix = np.arange(16*16).reshape(16,16)

        x_size = 8

        
        lst = []
        for y in range(0,len(color_matrix)-x_size+1,x_size):
            for k in range(0,len(color_matrix[0])-x_size+1,x_size):
                lst.append(color_matrix[y:y+x_size,k:k+x_size])
        
        C = np.array(lst)
        return C
    @staticmethod
    def get_quantization_matrix_luminance(Q = 50):
        S = 0
        if Q < 50:
            S = 5000/Q
        else:
            S = 200 -2*Q
        standart_quant_matrix = np.array([[16,11,10,16,24,40,51,61],
                                        [12,12,14,19,26,58,60,55],
                                        [14,13,16,24,40,57,69,56],
                                        [14,17,22,29,51,87,80,62],
                                        [18,22,37,56,68,109,103,77],
                                        [24,35,55,64,81,104,113,92],
                                        [49,64,78,87,103,121,120,101],
                                        [72,92,95,98,112,100,103,99]])
        Qquant_matrix = np.array(np.floor((S*standart_quant_matrix+50)/100),dtype=np.uint8)
        Qquant_matrix[Qquant_matrix  <= 0] =  1 
        return Qquant_matrix
    
    def get_quantization_matrix_chrominance(Q = 50):
        S = 0
        if Q < 50:
            S = 5000/Q
        else:
            S = 200 -2*Q
        standart_quant_matrix = np.array([
                                        [17,18,24,47,99,99,99,99],
                                        [18,21,26,66,99,99,99,99],
                                        [24,26,56,99,99,99,99,99],
                                        [47,66,99,99,99,99,99,99],
                                        [99,99,99,99,99,99,99,99],
                                        [99,99,99,99,99,99,99,99],
                                        [99,99,99,99,99,99,99,99],
                                        [99,99,99,99,99,99,99,99]])
        Qquant_matrix = np.array(np.floor((S*standart_quant_matrix+50)/100),dtype=np.uint8)
        Qquant_matrix[Qquant_matrix  <= 0] =  1 
        return Qquant_matrix
    
    
    @staticmethod
    def quantize_matrix(DCP_matrix_8x8:np.ndarray,Qquant_matrix):
        result_matrix = DCP_matrix_8x8/Qquant_matrix
        return result_matrix.round()
    
    @staticmethod
    def dequantize_matrix(quantized_matrix,Qquant_matrix):
        result_matrix = quantized_matrix*Qquant_matrix

        return result_matrix
    
    @staticmethod
    def encode_image(filename_format: str, quality : int,out_filename):
        wi = workImage(filename_format)
        wi.rgb_to_YCbCr()
        
        img_x,img_y = wi.img_size

        q_lum = JPEG.get_quantization_matrix_luminance(quality)
        q_chrom = JPEG.get_quantization_matrix_chrominance(quality)
        B = DCT.get_B_matrix()
        
        Y_channel = wi.ycbcr_img_arr[:,:,0]
        Cb_channel = workImage.downSampling_channel(wi.ycbcr_img_arr[:,:,1],2,2,'average')
        Cr_channel = workImage.downSampling_channel(wi.ycbcr_img_arr[:,:,2],2,2,'average')
        
        
     
        block_matrix_y = JPEG.split_to_8_8_blocks(Y_channel)
        block_matrix_cb = JPEG.split_to_8_8_blocks(Cb_channel)
        block_matrix_cr = JPEG.split_to_8_8_blocks(Cr_channel)
    
    
        
    
        cb_cr_img_x = int(img_x/2)
        cb_cr_img_y = int(img_y/2)
    
        re_block_matrix_y = JPEG.encode_blocks(img_x,img_y,block_matrix_y,B,q_lum)
        re_block_matrix_cb_downScaled = JPEG.encode_blocks(cb_cr_img_x,cb_cr_img_y,block_matrix_cb,B,q_chrom)
        re_block_matrix_cr_downScaled = JPEG.encode_blocks(cb_cr_img_x,cb_cr_img_y,block_matrix_cr,B,q_chrom)
        
        
        encoded_block_y = np.array(JPEG.RLE_encode(re_block_matrix_y))
        encoded_block_cb = np.array(JPEG.RLE_encode(re_block_matrix_cb_downScaled))
        encoded_block_cr = np.array(JPEG.RLE_encode(re_block_matrix_cr_downScaled))

        
        pack_lst = [img_x,img_y,quality,encoded_block_y,encoded_block_cb,encoded_block_cr]
        
        
        with open(out_filename +'.com_jpeg','wb') as w_file:
            pickle.dump(pack_lst,w_file)
        
        pass
    

   
    @staticmethod
    def encode_blocks(img_x,img_y,blocks_matrix,B,q):
        

        
        zig_zag_blocks_array = np.zeros((int(img_x*img_y/64),64),dtype= np.int16)
        zig_zag_index = 0

        
        for block in blocks_matrix:
            
            DCT_block = DCT.fast_FDCT(block,B)
            quantized = JPEG.quantize_matrix(DCT_block,q)
            

            #* ZIG_ZAG_SCANNING  
            zig_zig_scanned = workImage.zig_zag_scanning(quantized)
            zig_zag_blocks_array[zig_zag_index] = zig_zig_scanned
            zig_zag_index+=1
            
            
        
        #* ОБЪЕДИНЕНИЕ БЛОКОВ В ПОСЛЕДОВАТЕЛЬНОСТЬ            
        zig_zag_combined = zig_zag_blocks_array.reshape((img_x*img_y,1))    
        
        return zig_zag_combined

    @staticmethod
    def RLE_encode(block_sequence : np.ndarray) -> list:
        '''
        Функция кодирования квантованых блоков 
        
        in: block_sequence – последовательность квантованых коэффициентов вида (img_x*img_y,1)
        (внутри каждого элемента массив из одного элемента)
        out: list – вида [значение,кол-во нулей, значение, кол-во нулей и т.д. ]
        '''
         
        lst = []
        null_counter = 0
        lst.append(block_sequence[0][0])
        
        for q_index in range(1,len(block_sequence)):
            if block_sequence[q_index][0] != 0:
                lst.append(null_counter)
                lst.append(block_sequence[q_index][0])
                null_counter = 0
            else:
                null_counter += 1 
        
        if null_counter != 0:
            lst.append(null_counter)
        
        return lst
         
    @staticmethod
    def RLE_decode(encoded_sequence: list) -> list:
        decoded_lst = []
        for k in range(len(encoded_sequence)):
            if k % 2 ==0:
                decoded_lst.append(encoded_sequence[k])
            else:
                iter = 0
                while encoded_sequence[k] > iter:
                    decoded_lst.append(0)
                    iter+=1
        return np.array(decoded_lst)
    
    def decode_blocks(decoded_block_seq,img_x,img_y,q_matrix,B):
        num_of_blocks = int(img_x*img_y/64)

        lst1 = [[] for _ in range(img_y)]
        ind_x = 0
        ind_y = 0

        for k in range(num_of_blocks):
            de_zig_zag = workImage.zig_zag_writiing(decoded_block_seq[k*64:(k+1)*64],8,8)
            de_DCT_block = JPEG.dequantize_matrix(de_zig_zag,q_matrix)
            re_block_matrix = DCT.fast_IDCT(de_DCT_block,B)

            re_block_matrix[re_block_matrix > 255] = 255
            re_block_matrix[re_block_matrix < 0] = 0
            
            ind_line = 0
            for line in re_block_matrix:
                if ind_x + 8 > img_x:
                    ind_x = 0
                    ind_y += 8
                lst1[ind_y + ind_line].append(line)
                ind_line+=1
            ind_x+=8
        
        for k in range(len(lst1)):
            lst1[k] = np.concatenate(lst1[k])
        

        return np.array(lst1,dtype=np.uint8)
        
    @staticmethod
    def decode_Image(encoded_filename):
        with open(encoded_filename+'.com_jpeg','rb') as r_file:
            packed_lst = pickle.load(r_file)
        img_x,img_y,quality,encoded_blocks_y,encoded_blocks_cb,encoded_blocks_cr = packed_lst
        
        quant_lum = JPEG.get_quantization_matrix_luminance(quality)
        quant_chrom = JPEG.get_quantization_matrix_chrominance(quality)
        
        B = DCT.get_B_matrix()
        
        decoded_blocks_y = JPEG.RLE_decode(encoded_blocks_y)
        decoded_blocks_cb = JPEG.RLE_decode(encoded_blocks_cb)
        decoded_blocks_cr = JPEG.RLE_decode(encoded_blocks_cr)
        
        chrom_x = int(img_x/2)
        chrom_y = int(img_y/2)
        
        
        decoded_Y = JPEG.decode_blocks(decoded_blocks_y,img_x,img_y,quant_lum,B)
        decoded_cb = JPEG.decode_blocks(decoded_blocks_cb,chrom_x,chrom_y,quant_chrom,B)
        decoded_cr = JPEG.decode_blocks(decoded_blocks_cr,chrom_x,chrom_y,quant_chrom,B)
        
        upscaled_cb =  workImage.upSampling_channel(decoded_cb,2,2)
        upscaled_cr =  workImage.upSampling_channel(decoded_cr,2,2)
        
        end_img = np.dstack((decoded_Y,upscaled_cb,upscaled_cr))
        img = Image.fromarray(end_img,mode='YCbCr')
        rgb_img = img.convert(mode='RGB')
        rgb_img.save(encoded_filename+'.png')
class DCT:
    @staticmethod
    def get_coef(coef_ind:int):
        if coef_ind == 0:
            return 1/np.sqrt(2)
        else:
            return 1
    
    @staticmethod
    def Forward_Transfromation(color_matrix_8x8:np.ndarray) -> np.ndarray:
        '''Функция прямого дискретного косинусного преобразования массива 8 на 8
        РАБОТАЕТ НЕКОРРЕКТНО 
        Входные данные: матрица 8 на 8
        Выходные данные: матрица 8 на 8 коэффициетов ДКП
        '''
        
        
        
        
        out_matrix = np.zeros((8,8),dtype=np.int16) #!Создание матрицы на выходе
        shift = 2**(8-1) -1 
        color_matrix_8x8 = color_matrix_8x8 - shift #!Смещение значений
        #! Преобразование
        for u in range(8):
            for v in range(8):
                out_matrix[v,u] = 1/4 * DCT.get_coef(v)*DCT.get_coef(u) * np.sum(
                    [[color_matrix_8x8[y,x] * np.cos((2*x+1)*np.pi*u/16) * np.cos((2*y+1)*np.pi*v/16) for y in range(8)] for x in range(8)]
                )
        
        return out_matrix
    
    def Inverse_Transformation(DCT_matrix_8x8: np.ndarray) -> np.ndarray:
        '''
        РАБОТАЕТ НЕКОРРЕКТНО 
        Функция обратного преобразования дискретного косинусного преобразования
        
        Входные данные: матрица 8 на 8 коэффициентов ДКП
        Выходные данные: исходная матрица 8 на 8 
        '''
        
        
        
        out_matrix = np.zeros((8,8)) #!Создание матрицы на выходе
        shift = 2**(8-1) -1 
        
        for x in range(8):
            for y in range(8):
                out_matrix[y,x] = 1/4 * np.sum([[DCT.get_coef(v)* DCT.get_coef(u) * DCT_matrix_8x8[v,u] * 
                                                 np.cos((2*x+1)*np.pi*u/16) * np.cos((2*y+1)*np.pi*v/16) for v in range(8)] for u in range(8)]
                )
        out_matrix = out_matrix + shift
        out_matrix[out_matrix >255] = 255
        out_matrix[out_matrix < 0] = 0
        return out_matrix.round()
    
    @staticmethod
    def get_B_matrix():
        B = np.zeros((8,8))
        for i in range(8):
            for j in range(8):
                if i ==0 :
                    B[i,j] = np.sqrt(1/8)
                else:
                    B[i,j] = np.sqrt(2/8) * np.cos((2*j+1) * i * np.pi/(2*8))
        return B
    
    @staticmethod
    def fast_FDCT(color_matrix_8x8:np.ndarray,B : np.ndarray) -> np.ndarray:
        '''
        Функция прямого дискретного преобразования, основанная на матричном умножении
        
        Входные данные: матрица 8 на 8
        Выходные данные: матрица 8 на 8 коэффициетов ДКП
        '''




        return np.matmul(np.matmul(B,color_matrix_8x8),B.transpose())
    @staticmethod
    def fast_IDCT(DCT_matrix_8x8: np.ndarray,B:np.ndarray) -> np.ndarray:
        '''
        Функция обратного преобразования дискретного косинусного преобразования, основанного на матричном умножении
        
        Входные данные: матрица 8 на 8 коэффициентов ДКП
        Выходные данные: исходная матрица 8 на 8 
        '''
        return np.matmul(np.matmul(B.transpose(),DCT_matrix_8x8),B)
        
    





pass
# # mat = np.zeros((5,5),np.uint8)

# # counter = 1
# # for i in range(5):
# #     for j in range(5):
# #         mat[i][j] = counter
# #         counter+=1


# ret = workImage.upSampling(wi.rgb_img_arr,2,2)
# img = Image.fromarray(ret)
# print(img.size)
# img.show()

# ycbcr = wi.rgb_to_YCbCr()
# img = Image.fromarray(ycbcr)
# img.show()
# rgb = wi.YCbCr_to_rgb(ycbcr)
# img = Image.fromarray(rgb)
# img.show()



def gen_norm_image(name):

    generator = np.random.default_rng()
    r = generator.integers(0,255,512*512,dtype=np.uint8,endpoint=True).reshape((512,512))
    g = generator.integers(0,255,512*512,dtype=np.uint8,endpoint=True).reshape((512,512))
    b = generator.integers(0,255,512*512,dtype=np.uint8,endpoint=True).reshape((512,512))

    rgb = np.dstack((r,g,b))
    image = Image.fromarray(rgb)
    image.show()
    image.save(name)

pass


# JPEG.encode_image('lena.tif',20,'lena')
# JPEG.decode_Image('lena')



#JPEG.encode_image('nia.jpg',50)




filename = 'rgb.png'
filecut = 'red'

JPEG.decode_Image(filecut + '1')

# for k in range(10,101,10):
#     JPEG.decode_Image(filecut + str(k))
#wi = workImage('rgb.png')
#wi.rgb_to_YCbCr()

# JPEG.encode_image('rgb.png',50,'rgb')
# JPEG.decode_Image('rgb')
# pack = [wi.rgb_img_arr[:,:,0],wi.rgb_img_arr[:,:,1],wi.rgb_img_arr[:,:,2]]
# with open('rgb_original.com_jpeg','wb') as write_F:
#     pickle.dump(pack,write_F)



