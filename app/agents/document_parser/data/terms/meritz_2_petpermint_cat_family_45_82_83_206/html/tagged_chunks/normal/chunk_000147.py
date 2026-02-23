from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험계약대출은 순수보장성 상품 등 보험상<br>품의 종류 및 보험계약 경과기간에 따라 제한 될 수 있<br>습니다.</p><h1 '
 "id='1' style='font-size:20px'>제22조(계약의 무효)</h1><br><p id='2' "
 "data-category='paragraph' style='font-size:16px'>\uf000 다음 중 한 가지에 해당하는 경우에는 "
 '계약을 무효로 하<br>며 이미 납입한 보험료를 돌려드립니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
