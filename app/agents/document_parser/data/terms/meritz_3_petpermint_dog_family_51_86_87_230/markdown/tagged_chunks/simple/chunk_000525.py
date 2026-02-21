from langchain_core.documents import Document

chunk = Document(
    page_content=('- 용\n'
 '- ㉡ 피보험자가「배상책임 관련 특별약관 일반조항」제8\n'
 '- 조(손해방지의무)의 제1항 제2호의 제3자로부터 손\n'
 '- 해의 배상을 받을 수 있는 그 권리를 지키거나 행사\n'
 '- 하기 위하여 지출한 필요 또는 유익하였던 비용\n'
 '- ㉢ 피보험자가 지급한 소송비용, 변호사비용, 중재, 화\n'
 '- 해 또는 조정에 관한 비용\n'
 '- ㉣ 보험증권상 보상한도액내의 금액에 대한 공탁보증보\n'
 '- 험료. 그러나 회사는 그러한 보증을 제공할 책임은\n'
 '186# 부담하지 않습니다.㉤ 피보험자가「배상책임 관련 특별약관 일반조항」제9'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000525',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
