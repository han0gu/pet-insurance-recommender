from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[심신상실자(心神喪失者)]\n'
 '의식은 있으나 장애의 정도가 심하여 자신의 행위 결과를 합리적으로 판단할 능력을 갖지 못한 사 람을 말합니다. [심신박약자(心神薄弱者)] '
 '심신상실의 상태까지는 이르지 않았으나, 마음이나 정신의 장애로 인하여 사물을 변별할 능력이나 의사를 결정할 능력이 미약한 사람을 '
 '말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
