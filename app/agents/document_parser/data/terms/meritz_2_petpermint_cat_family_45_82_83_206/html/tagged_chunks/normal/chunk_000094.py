from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보험료 갱신형 계약 등 일부 보험계약의 경우<br>분납이 제한될 수 있습니다.</p><br><h1 id='35' "
 "style='font-size:20px'>【위험변경시 해약환급금 정산】</h1><br><p id='36' "
 "data-category='paragraph' style='font-size:20px'>제1항에 따라 위험이 증가ㆍ감소되는 경우 이후 "
 '기간 보<br>장을 위한 재원인 계약자적립액 등의 차이로 계약자가 추<br>가로 납입하여야 할(또는 반환받을) 금액이 발생할 수 '
 "있<br>습니다.</p><h1 id='37'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
