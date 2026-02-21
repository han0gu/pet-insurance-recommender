from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피해자로부터 손해배상청구를 받았을 경우<br>3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우</p><br><p '
 "id='40' data-category='paragraph' style='font-size:14px'>② 계약자 또는 피보험자가 제1항 "
 '각호의 통지를 게을리하여 손해가 증가된 때에는 회사<br>는 그 증가된 손해를 보상하여 드리지 않으며, 제1항제3호의 통지를 게을리 한 '
 '때에는<br>소송비용과 변호사비용도 보상하여 드리지 않습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
