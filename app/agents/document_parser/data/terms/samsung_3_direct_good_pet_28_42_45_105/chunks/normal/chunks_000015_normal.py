from langchain_core.documents import Document

chunk = Document(
    page_content='. 그러나 그 후유장해가 이미 상해 후유장해(80% 이상) 보험금을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 상',
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 29},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['joint', 'head', 'other']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 74,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
