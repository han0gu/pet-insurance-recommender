from langchain_core.documents import Document

chunk = Document(
    page_content=("id='15' data-category='paragraph' style='font-size:20px'>\uf000 계약자는 보장개시일부터 "
 '2년 이상 지난 유효한 계약으<br>로서 계약자의 요청이 있는 경우에 한하여 보험연도 기준<br>연4회에 한하여 중도인출 할 수 '
 "있습니다.</p><br><p id='16' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제1항의 중도인출금은 계약자가 요청한 시점에서 계산된<br>기본계약 해약환급금과 "
 '기본계약 적립부분 해약환급금 중<br>적은 금액의 80% 범위'),
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
 'indexing': {'chunk_id': 'chunk_000230',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
