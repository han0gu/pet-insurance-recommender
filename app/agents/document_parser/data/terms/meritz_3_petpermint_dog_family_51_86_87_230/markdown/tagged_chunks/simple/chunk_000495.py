from langchain_core.documents import Document

chunk = Document(
    page_content=('금액을 결정합니다.\n'
 '\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한\n'
 '경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을\n'
 '미치지 않습니다.# 제8조(손해방지의무)\uf000 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의\n'
 '사항을 이행하여야 합니다.① 손해의 방지 또는 경감을 위하여 노력하는 일(피해자178에 대한 응급처치, 긴급호송 또는 그 밖의 '
 '긴급조치를\n'
 '포함합니다)- ② 제3자로부터 손해의 배상을 받을 수 있는 경우에는 그\n'
 '- 권리의 보전 또는 행사를 위한 필요한 조치를 취하는\n'
 '- 일'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000495',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
