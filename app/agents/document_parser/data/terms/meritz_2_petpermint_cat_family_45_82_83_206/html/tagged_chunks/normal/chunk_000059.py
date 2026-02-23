from langchain_core.documents import Document

chunk = Document(
    page_content=('[보장]공시이율이 0.1%인 경우, 계약자적립액은<br>[보장]공시이율(0.1%)이 아닌 최저보증이율(0.3%)로 '
 "적<br>립됩니다.</p><h1 id='79' style='font-size:20px'>【운용자산이익률】</h1><br><p "
 "id='80' data-category='paragraph' style='font-size:16px'>직전 1년간의 운용자산에 대한 "
 "투자영업수익과 투자영<br>업비용 등을 고려하여 산출</p><br><h1 id='81' "
 "style='font-size:20px'>【외부지표금리】</h1><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000059',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
