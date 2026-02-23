from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또는 척추측만증(척추가 옆으로 휘어지는 증상) 변형이 있을 때\n'
 '- 나) 척추체(척추뼈 몸통) 한 개의 압박률이 20%이상인 경우 또는 한 운동단위\n'
 '- 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈 몸\n'
 '- 통)의 압박률의 합이 40% 이상일 때\n'
 '- 12) "추간판탈출증으로 인한 심한 신경 장해" 란 추간판탈출증으로 추간판을 2마\n'
 '- 디 이상(또는 1마디 추간판에 대해 2회 이상) 수술하고도 마미신경증후군이\n'
 '- 발생하여 하지의 현저한 마비 또는 대소변의 장해가 있는 경우'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000779',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
