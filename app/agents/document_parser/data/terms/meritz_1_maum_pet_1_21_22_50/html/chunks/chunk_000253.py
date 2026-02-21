from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 피해자에 대하여 부담하는 법률상의 손해배상책임액이 보험증권에 기재<br>된 보상한도액을 명백하게 초과하는 때<br>2. '
 "피보험자가 정당한 이유없이 협력하지 않은 때</p><br><p id='84' data-category='paragraph' "
 "style='font-size:14px'>⑤ 회사가 제1항의 절차를 대행하는 경우에는, 피보험자에 대하여 보상책임을 지는 "
 '한도<br>내에서, 가압류나 가집행을 면하기 위한 공탁금을 피보험자에게 대부할 수 있으며 이에<br>소요되는 비용을 보상합니다'),
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
