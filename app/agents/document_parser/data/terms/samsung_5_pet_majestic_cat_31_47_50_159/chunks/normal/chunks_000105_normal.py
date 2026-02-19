from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[심신상실자(心神喪失者)]\n'
 '의식은 있으나 장애의 정도가 심하여 자신의 행위 결과를 합리적으로 판단할 능력을 갖지 못한 사 람을 말합니다.\n'
 '[심신박약자(心神薄弱者)]\n'
 '심신상실의 상태까지는 이르지 않았으나, 마음이나 정신의 장애로 인하여 사물을 변별할 능력이나 의사를 결정할 능력이 미약한 사람을 '
 '말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000105',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
