from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자(타인을 위한 계약의 경우 그 특정된 타인을 포함합니다)에게 다음의 내용을 서면(등기우편\n'
 '등), 전화(음성녹음) 또는 전자문서 등으로 알려드립니다. 다만, 계약이 해지되기 전에 발생한 보험\n'
 '금 지급사유에 대하여 회사는 계약상의 보장을 합니다.- 1. 납입최고(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용\n'
 '- 2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우그 끝나는 날의 다음날에 계\n'
 '- 약이 해지된다는 내용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
