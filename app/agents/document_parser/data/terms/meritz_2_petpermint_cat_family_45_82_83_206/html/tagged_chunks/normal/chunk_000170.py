from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>생년월일 : 1988년 10월 2일<br>현재(계약일) : 2023년 4월 "
 "14일</p><br><p id='29' data-category='paragraph' style='font-size:18px'>⇒ "
 "2023년 4월 14일 - 1988년 10월 2일</p><br><h1 id='30' style='font-size:18px'>= 34년 "
 "6월 12일 = 35세</h1><h1 id='31' style='font-size:18px'>【 계약해당일 】</h1><br><p "
 "id='32'"),
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
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
