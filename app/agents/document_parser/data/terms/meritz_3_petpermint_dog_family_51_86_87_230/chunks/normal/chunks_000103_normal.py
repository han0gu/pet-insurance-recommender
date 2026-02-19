from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 전자문서에 보험금 지급사유, 보험금액, 보험계약자와 보험수익자의 신원, 보험기간이 적혀 있을 것 2. 전자문서에 법 '
 '제731조제1항에 따른 전자서명(이하 “전자서명”이라 한다)을 하기 전에 전자서명을 할 사람을 직접 만나서 전자서명을 하는 사람이 '
 '보험계약 에 동의하는 본인임을 확인하는 절차를 거쳐 작성될 것 3. 전자문서에 전자서명을 한 후에 그 전자서명을 한 사 람이 보험계약에 '
 '동의한 본인임을 확인할 수 있도록 지문정보를 이용하는 등 법무부장관이 고시하는 요건 을 갖추어 작성될 것 4'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 72},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000103',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
