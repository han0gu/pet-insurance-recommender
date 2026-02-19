from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 주1) 안전수동 : 물체를 감별할 정도의 시력상태 가 아니며 눈앞에서 손의 움직임을 식별할 수 있을 정도의 시력상태 주2) 안전수지 '
 ': 시표의 가장 큰 글씨를 읽을 수 있는 정도의 시력은 아니나 눈 앞 30cm 이내 에서 손가락의 개수를 식별할 수 있을 정도 의 '
 '시력상태\n'
 '5) 안구(눈동자) 운동장해의 판정은 질병의 진단 또는 외상 후 1년 이상이 지난 뒤에 그 장해정도를 평가한다. 6) “안구(눈동자)의 '
 '뚜렷한 운동장해”라 함은 아래의 두 경우 중 하나에 해당하는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000630',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
