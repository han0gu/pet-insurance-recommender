from langchain_core.documents import Document

chunk = Document(
    page_content=("소멸)</h1><br><p id='35' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제3조(보험금의 지급사유)에서 정한 일반상해80%이상후<br>유장해보험금 지급사유가 "
 '발생한 경우에는 이 보장책임은<br>그 때부터 소멸됩니다.<br>\uf000 제1항에 따라 이 계약의 보장책임이 소멸된 때에는 '
 '회사<br>는 이 보장책임의 해약환급금을 지급하지 않으며, 그 때까<br>지「보험료 및 해약환급금 산출방법서」에서 정하는 바에<br>따라 '
 '회사가 적립한 적립부분의 계약자적립액(중도인출이<br>있는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
