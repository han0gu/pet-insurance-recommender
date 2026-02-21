from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급\n'
 '제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여\n'
 '피보험자 또는 보험수익자에게 즉시 통지합니다. 다만, 지\n'
 '급예정일은 다음 각 호의 어느 하나에 해당하는 경우를 제\n'
 '외하고는 제7조(보험금의 청구)에서 정한 서류를 접수한 날\n'
 '부터 30영업일 이내에서 정합니다.- ① 소송제기\n'
 '- ② 분쟁조정 신청\n'
 '- ③ 수사기관의 조사\n'
 '- ④ 해외에서 발생한 보험사고에 대한 조사\n'
 '- ⑤ 제6항에 따른 회사의 조사요청에 대한 동의 거부 등'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000024',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
