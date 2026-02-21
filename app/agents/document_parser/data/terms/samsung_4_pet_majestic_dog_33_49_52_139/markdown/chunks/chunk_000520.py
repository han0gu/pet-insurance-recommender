from langchain_core.documents import Document

chunk = Document(
    page_content=('(독촉)기간으로 정하여 계약자에게 다음 각 호의 내용을 서면(등기우편 등), 전화(음성\n'
 '녹음) 또는 전자문서 등으로 알려 드립니다.- 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용\n'
 '- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고(독촉)\n'
 '- 기 간이 끝나는 날의 다음날에 특별약관이 해지된다는 내용\n'
 '- 3. 계약자가 회사로부터 보험계약대출을 받은 경우 특별약관이 해지되는 즉시 해약환\n'
 '- 급금에서 보험계약대출원금과 이자가 차감된다는 내용'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
