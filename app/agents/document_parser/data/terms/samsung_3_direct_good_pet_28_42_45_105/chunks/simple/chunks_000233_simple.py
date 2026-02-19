from langchain_core.documents import Document

chunk = Document(
    page_content=('52 / 181\n'
 '의 서면(「전자서명법」제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행 령 제44조의2에 정하는 바에 따라 본인 확인 및 '
 '위조·변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함)에 의한 동의를 얻지 않은 경우. 다만, 단체가 규약에 따라 구성원의 전부 또는 '
 '일부를 피보험자로 하는 계약을 체결하는 경우에는 이를 적용 하지 않습니다. 이 때 단체보험의 보험수익자를 피보험자 또는 그 상속인이 아닌 '
 '자로 지정할 때에는 단체의 규약에서 명시적으로 정한 경우가 아니면 이를 적용합 니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 53},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000233',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
