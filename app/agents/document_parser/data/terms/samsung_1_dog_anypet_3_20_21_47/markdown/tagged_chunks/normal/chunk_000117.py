from langchain_core.documents import Document

chunk = Document(
    page_content=('- 간주하여 제1항에 의한 보상할 금액을 결정합니다.\n'
 '- ③ 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금\n'
 '- 결정에는 영향을 미치지 않습니다.\n'
 '# 제7조(손해방지의무)# ① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 사항을 이행하여야 합니다.- 1. 손해의 방지 또는 '
 '경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급호송 또는 그 밖의\n'
 '- 긴급조치를 포함합니다)\n'
 '- 2. 제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리를 지키거나 행사하기 위한 필요한'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
