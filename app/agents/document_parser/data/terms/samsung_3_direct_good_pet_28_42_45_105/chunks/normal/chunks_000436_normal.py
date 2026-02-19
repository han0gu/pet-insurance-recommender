from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 상속순위\n'
 '① 피상속인의 직계비속 ③ 피상속인의 형제자매\n'
 '② 피상속인의 직계존속 ④ 피상속인의 4촌 이내의 방계혈족\n'
 '[직계비속] 자기로부터 직계로 이어져 내려가는 혈족. 자녀, 손자, 증손 등 [직계존속] 조상으로부터 직계로 내려와 자기에 이르는 사이의 '
 '혈족. 부모, 조부모 등 [방계혈족] 자기의 형제자매와 형제자매의 직계비속, 직계존속의 형제자매 및 그 형제자매의 직계비속'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000436',
              'chunk_char_len': 211,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
