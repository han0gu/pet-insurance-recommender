from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 【붙임1】(특정신체부위 분류표) 중에서 회사가 지정한 부위(이하「특정신체부위」 라 합니다)에 발생한 질병 또는 특정신체부위에 발생한 '
 '질병의 전이로 인하여 특정 신체부위 이외의 부위에 발생한 질병(단, 전이는 합병증으로 보지 않습니다) 2. 【붙임2】(특정질병 분류표) '
 '중에서 회사가 지정한 질병(이하「특정질병」이라 합 니다)\n'
 '<용어풀이>\n'
 '[장해지급률]\n'
 '질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로 나타낸 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 129},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000825',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
