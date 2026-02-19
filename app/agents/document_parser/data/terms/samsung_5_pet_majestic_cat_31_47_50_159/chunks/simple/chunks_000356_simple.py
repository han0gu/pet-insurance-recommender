from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합니다) 중에 상해를 입고 그 직접적인 결과로써 '
 '[별표-상해관련2]골절(치아 파절(깨 짐, 부러짐) 제외) 분류표에 정한 골절(이하「골절」이라 합니다)로 진단확정된 경우 에는 보험증권에 '
 '기재된 이 특별약관의 보험가입금액을 골절 진단비(치아 파절(깨짐, 부러짐) 제외)로 보험수익자에게 지급합니다. ② 제1항의 골절 '
 '진단비(치아 파절(깨짐, 부러짐) 제외)는 매사고마다 지급합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 69},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000356',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
