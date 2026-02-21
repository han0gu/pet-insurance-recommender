from langchain_core.documents import Document

chunk = Document(
    page_content=(". 대위권: 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.</p><br><p id='20' "
 "data-category='paragraph' style='font-size:14px'>② 제1항에서 정의되지 않은 용어는 보통약관 "
 "제2조(용어의 정의)를 따릅니다.</p><h1 id='21' style='font-size:14px'>제3조(보상하는 "
 "손해)</h1><br><p id='22' data-category='list' style='font-size:14px'>① 회사는 "
 '피보험자가 대한민국 내에서 이 특별약관의 보험기간'),
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
