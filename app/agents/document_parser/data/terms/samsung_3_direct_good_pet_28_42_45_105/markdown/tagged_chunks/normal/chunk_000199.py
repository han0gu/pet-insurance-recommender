from langchain_core.documents import Document

chunk = Document(
    page_content=('령 제44조의2에 정하는 바에 따라 본인 확인 및 위조·변조 방지에 대한 신뢰성을\n'
 '갖춘 전자문서를 포함)에 의한 동의를 얻지 않은 경우. 다만, 단체가 규약에 따라\n'
 '구성원의 전부 또는 일부를 피보험자로 하는 계약을 체결하는 경우에는 이를 적용\n'
 '하지 않습니다. 이 때 단체보험의 보험수익자를 피보험자 또는 그 상속인이 아닌\n'
 '자로 지정할 때에는 단체의 규약에서 명시적으로 정한 경우가 아니면 이를 적용합\n'
 '니다.2. 만 15세 미만자, 심신상실자 또는 심신박약자를 피보험자로 하여 사망을 보험금 지'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
