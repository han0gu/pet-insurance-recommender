from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제1관 목적 및 용어의 정의# 제 1조 (목적)이 보험계약(이하「계약」이라 합니다)은 보험계약자(이하「계약자」라 합니다)와 보험회\n'
 '사(이하「회사」라 합니다) 사이에 피보험자의 질병이나 상해에 대한 위험을 보장하기 위\n'
 '하여 체결됩니다.# 제 2조 (용어의 정의)이 계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항에서 달리 정의되지 않는 한\n'
 '다음과 같습니다.# ① 계약관계 관련 용어- 1. 계약자: 회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
