from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항에서 정한 특별약관의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피보험\n'
 '자가 최초계약 청약시 제11조(계약 전 알릴 의무)를 위반한 경우에는 제13조(알릴 의-'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000369',
              'chunk_char_len': 97,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
