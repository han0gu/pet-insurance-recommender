from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우(다만, 전동휠체어, 의료용 스쿠터 등 보행보조용 의자차는 제외합니다.)\n'
 '② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 제23조(특별약관 내용\n'
 '의 변경 등)에 따라 특별약관 내용을 변경할 수 있습니다.[위험변경에 따른 계약변경 절차]<유의사항>위험변경사항 통지(우편, 전화, 방문 '
 '등)![image](/image/placeholder)\n'
 '↓계약자, 피보험자의 계약변경사항 확인 후 청약![image](/image/placeholder)\n'
 '↓계약변경사항 인수 심사| ↓ |\n'
 '| --- |'),
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
