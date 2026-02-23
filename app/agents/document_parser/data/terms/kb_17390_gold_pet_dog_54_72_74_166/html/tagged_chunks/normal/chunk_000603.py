from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하<br>며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 '전액 부담합니다.</p><br><h1 id=\'111\' style=\'font-size:16px\'>제3조("창상봉합술(급여)"의 '
 "정의)</h1><br><p id='112' data-category='paragraph' "
 'style=\'font-size:16px\'>\uf000 이 특별약관에 있어서 "창상봉합술(급여)"이라 함은 상해의 직접결과로써, '
 '"창상</p><br><h1 id=\'113\''),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000603',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
