from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 금융회사(우체국을 포함합니다)를<br>통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서<br>류를 영수증으로 '
 "대신합니다.</p><br><h1 id='48' style='font-size:20px'>【납입기일】</h1><br><p id='49' "
 "data-category='paragraph' style='font-size:16px'>계약자가 제2회 이후의 보험료를 납입하기로 한 "
 "날을 말합<br>니다.</p><footer id='50' style='font-size:14px'>71</footer><p id='51'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000184',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
