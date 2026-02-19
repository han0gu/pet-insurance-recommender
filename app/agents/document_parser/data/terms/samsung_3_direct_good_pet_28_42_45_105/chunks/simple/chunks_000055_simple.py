from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러지 않을 경우 보험사고가 발생한 경우에도 보험금 지급이 제한될 수 있습니다. 원동기장치 자전거는 전동킥보드, 전동이륜평행차, '
 '전동기의 동력만으로 움직일 수 있는 자전거 등 개인형 이동장치를 포함하며, 장애인 또는 교통약자가 사용하는 보행보조용 의자차인 '
 '전동휠체어, 의료용 스쿠터 등은 제외됩니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
