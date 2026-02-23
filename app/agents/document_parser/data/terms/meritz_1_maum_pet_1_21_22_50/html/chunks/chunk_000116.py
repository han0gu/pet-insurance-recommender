from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 그 보험금 지급사<br>유가 발생한 사실을 알지 못한 경우에는 청약철회의 효력은 발생하지 않습니다.<br>⑥ 제1항에서 '
 "보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야 합<br>니다.</p><h1 id='16' "
 "style='font-size:14px'>제21조(약관 교부 및 설명의무 등)</h1><br><p id='17' "
 "data-category='paragraph' style='font-size:14px'>① 회사는 계약자가 청약할 때에 계약자에게 약관의 "
 '중요한 내용을 설명하여야 하며,'),
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
