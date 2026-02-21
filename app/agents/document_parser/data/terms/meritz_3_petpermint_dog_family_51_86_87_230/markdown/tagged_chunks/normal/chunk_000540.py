from langchain_core.documents import Document

chunk = Document(
    page_content=('간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고(독\n'
 '촉)기간(납입최고(독촉)기간의 마지막 날이 영업일이 아닌\n'
 '때에는 최고(독촉)기간은 그 다음 날까지로 합니다)으로 정\n'
 '하여 계약자(타인을 위한 보험계약의 경우 특정된 보험수익\n'
 '자를 포함합니다)가 납입최고(독촉)기간 안에 보험료를 납\n'
 '입하지 않은 경우에는 납입최고(독촉)기간이 끝나는 날의\n'
 '다음날에 이 보장계약을 해제합니다.\n'
 '\uf000 다만, 해제 전에 발생한 보험금 지급사유에 대하여 회사\n'
 '는 보상합니다. 이 경우 계약자는 즉시 갱신보장 보험료를'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000540',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
