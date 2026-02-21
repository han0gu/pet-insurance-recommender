from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 피해자로부터 손해배상청구를 받았을 경우\n'
 '3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우② 계약자 또는 피보험자가 제1항 각호의 통지를 게을리하여 손해가 증가된 '
 '때에는 회사\n'
 '는 그 증가된 손해를 보상하여 드리지 않으며, 제1항제3호의 통지를 게을리 한 때에는\n'
 '소송비용과 변호사비용도 보상하여 드리지 않습니다. 다만, 계약자 또는 피보험자가 상\n'
 '법 제657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손'),
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
