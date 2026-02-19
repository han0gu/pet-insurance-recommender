from langchain_core.documents import Document

chunk = Document(
    page_content=('5) 개구장해는 턱관절의 이상으로 개구운동 제한이 있는 상태를 말하며, 최대 개구 상태에서 위·아래턱(상·하악)의 가운데 앞니(중절치)간 '
 '거리를 기준으로 한다. 단, 가운데 앞니(중절치)가 없는 경우에는 측정가능한 인접 치아간 거리의 최 대치를 기준으로 한다. 6) '
 '부정교합은 위턱(상악)과 아래턱(하악)의 부조화로 윗니(상악치아)와 아랫니(하 악치아)가 전방 및 측방으로 맞물림에 제한이 있는 상태를 '
 '말한다. 7) "말하는 기능에 심한 장해를 남긴 때" 라 함은 아래의 경우 중 하나 이상에 해 당되는 때를 말한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other', 'dental']},
 'indexing': {'chunk_id': 'chunk_000896',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
