from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>54</footer><h1 id='75' "
 "style='font-size:20px'>【[보장]공시이율】</h1><br><p id='76' "
 "data-category='paragraph' style='font-size:16px'>전통적인 보험상품에 적용되는 이율이 "
 '장기･고정금리이<br>기 때문에 시중금리가 급격하게 변동할 경우 이에 대응<br>하지 못하는 점을 고려하여, 시중의 지표금리 등에 '
 "연동<br>하여 일정기간마다 변동되는 이율을 말합니다.</p><h1 id='77'"),
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
 'indexing': {'chunk_id': 'chunk_000057',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
