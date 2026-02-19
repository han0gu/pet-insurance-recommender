from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자(보험수익자와 계약자가 다른 경우 보험수익자 를 포함합니다)에게 납입최고(독촉)기간 내에 연체보 험료를 납입하여야 한다는 내용 '
 '② 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하 지 않을 경우 납입최고(독촉)기간이 끝나는 날의 다음 날에 계약이 해지된다는 '
 '내용(이 경우 계약이 해지되 는 때에는 즉시 해약환급금에서 보험계약대출원금과 이자가 차감된다는 내용을 포함합니다)\n'
 '【 납입최고(독촉) 】\n'
 '약정된 기일까지 보험료가 납입되지 않을 경우, 회사가 계약자에게 납입을 재촉하는 것을 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000263',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
