from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 포함합니다)에게 납입최고(독촉)기간 내에 연체보\n'
 '- 험료를 납입하여야 한다는 내용\n'
 '- ② 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하\n'
 '- 지 않을 경우 납입최고(독촉)기간이 끝나는 날의 다음\n'
 '- 날에 계약이 해지된다는 내용(이 경우 계약이 해지되\n'
 '- 는 때에는 즉시 해약환급금에서 보험계약대출원금과\n'
 '- 이자가 차감된다는 내용을 포함합니다)\n'
 '# 【 납입최고(독촉) 】약정된 기일까지 보험료가 납입되지 않을 경우, 회사가\n'
 '계약자에게 납입을 재촉하는 것을 말합니다.\uf000 회사가 제1항에 따른 납입최고(독촉) 등을 전자문서로'),
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
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
