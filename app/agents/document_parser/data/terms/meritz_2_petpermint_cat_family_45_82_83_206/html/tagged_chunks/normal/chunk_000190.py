from langchain_core.documents import Document

chunk = Document(
    page_content=("포함) 등으로 계약자에게 안내하여 드립니다.</p><br><h1 id='53' "
 "style='font-size:20px'>【자동대출납입】</h1><br><p id='54' data-category='paragraph' "
 "style='font-size:16px'>보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대<br>출납입을 신청하면 해당 보험 상품의 "
 '해약환급금 범위 내<br>에서 납입할 보험료를 자동적으로 대출하여 이를 보험료<br>납입에 충당하는 서비스를 말합니다.</p><p '
 "id='55'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000190',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
