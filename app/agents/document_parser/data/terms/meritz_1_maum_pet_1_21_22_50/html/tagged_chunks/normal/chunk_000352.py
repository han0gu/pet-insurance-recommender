from langchain_core.documents import Document

chunk = Document(
    page_content=("위해 가족관계서류<br>의 수령을 생략할 수 있습니다.</p><footer id='23' style='font-size:14px'>- "
 "42 -</footer><h1 id='24' style='font-size:14px'>제5조(보험금 지급 등의 절차)</h1><br><p "
 "id='25' data-category='list' style='font-size:14px'>① 지정대리청구인은 제6조(보험금의 청구)에 "
 '정한 구비서류 및 제1조(적용대상)의 수익자<br>가 보험금을 직접 청구할 수 없는 특별한 사정이 있음을 증명하는 서류를'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000352',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
