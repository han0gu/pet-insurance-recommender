from langchain_core.documents import Document

chunk = Document(
    page_content=("날부터 1<br>개월 이내에 계약을 해지할 수 있습니다.</p><br><p id='60' data-category='list' "
 "style='font-size:16px'>① 계약자, 피보험자 또는 보험수익자가 보험금을 지급받<br>을 목적으로 고의로 보험금 "
 '지급사유를 발생시킨 경<br>우<br>② 계약자, 피보험자 또는 보험수익자가 보험금 청구에<br>관한 서류에 고의로 사실과 다른 것을 '
 '기재하였거나<br>그 서류 또는 증거를 위조 또는 변조한 경우'),
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
 'indexing': {'chunk_id': 'chunk_000404',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
