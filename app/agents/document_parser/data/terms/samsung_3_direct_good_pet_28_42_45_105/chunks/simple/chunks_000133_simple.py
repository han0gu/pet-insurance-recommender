from langchain_core.documents import Document

chunk = Document(
    page_content=('제34조 (보험계약대출)\n'
 '① 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정한 방법에 따라 대출(이하 「보 험계약대출」 이라 합니다)을 받을 수 있습니다. '
 '그러나 순수보장성보험 등 보험상품의 조르시 MU H 게약대추시 대차되 人□ 이시 INI'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
