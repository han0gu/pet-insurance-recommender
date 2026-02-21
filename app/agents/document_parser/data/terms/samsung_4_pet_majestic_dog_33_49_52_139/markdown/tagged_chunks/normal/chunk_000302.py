from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다) 중에 상해를 입고 그 직접적인 결과로써 [별표-상해관련2]골절(치아 파절(깨\n'
 '- 짐, 부러짐) 제외) 분류표에 정한 골절(이하「골절」이라 합니다)로 진단확정된 경우\n'
 '- 에는 보험증권에 기재된 이 특별약관의 보험가입금액을 골절 진단비(치아 파절(깨짐,\n'
 '- 부러짐) 제외)로 보험수익자에게 지급합니다.\n'
 '- ② 제1항의 골절 진단비(치아 파절(깨짐, 부러짐) 제외)는 매사고마다 지급합니다. 다만,\n'
 '- 동일한 상해사고를 직접적인 원인으로 2가지 이상의 골절 상태가 발생한 경우에는 1\n'
 '- 회에 한하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000302',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
