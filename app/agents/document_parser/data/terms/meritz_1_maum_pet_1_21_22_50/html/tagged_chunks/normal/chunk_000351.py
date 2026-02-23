from langchain_core.documents import Document

chunk = Document(
    page_content=('. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아<br>닌 경우에는 본인의 인감증명서, '
 '본인서명사실확인서 또는 안전성과 신뢰성이 확보된<br>전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)</p><br><p '
 "id='22' data-category='paragraph' style='font-size:14px'>② 회사는 제1항의 지정대리청구인 "
 "변경 지정시 계약자의 지정 편의를 위해 가족관계서류<br>의 수령을 생략할 수 있습니다.</p><footer id='23'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000351',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
