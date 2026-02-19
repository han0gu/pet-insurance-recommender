from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ 제1항의 규정에도 불구하고 다음 사항 중 어느 한 가지의 경우에 해당되는 사유로 보 험계약에서 정한 보험금의 지급사유가 발생한 경우 '
 '회사는 보험금을 지급하여 드리며, 보험료 납입면제사유가 발생한 경우 보험료 납입을 면제합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000656',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
