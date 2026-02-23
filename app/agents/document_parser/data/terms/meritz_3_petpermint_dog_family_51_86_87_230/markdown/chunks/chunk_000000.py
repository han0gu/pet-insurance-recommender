from langchain_core.documents import Document

chunk = Document(
    page_content=('제1관 목적 및 용어의 정의# 제1조(목적)이 보험계약(이하「계약」이라 합니다)은 보험계약자(이하\n'
 '「계약자」라 합니다)와 보험회사(이하「회사」라 합니다)\n'
 '사이에 피보험자의 상해에 대한 위험을 보장하기 위하여 체\n'
 '결됩니다.# 제2조(용어의 정의)이 계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항\n'
 '에서 달리 정의되지 않는 한 다음과 같습니다.# \uf000 계약 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 계약자 | 회사와 계약을 체결하고 보험료를 납입할 의 무를 지는 사람을 말합니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
