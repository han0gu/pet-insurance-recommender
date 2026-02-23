from langchain_core.documents import Document

chunk = Document(
    page_content=('- 대치를 기준으로 한다.\n'
 '- 6) 부정교합은 위턱(상악)과 아래턱(하악)의 부조화로 윗니(상악치아)와 아랫니(하\n'
 '- 악치아)가 전방 및 측방으로 맞물림에 제한이 있는 상태를 말한다.\n'
 '- 7) "말하는 기능에 심한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에 해\n'
 '- 당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 30% 미만인 경우\n'
 '- 나) 전실어증, 운동성실어증(브로카실어증)으로 의사소통이 불가한 경우\n'
 '- 8) "말하는 기능에 뚜렷한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000758',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
