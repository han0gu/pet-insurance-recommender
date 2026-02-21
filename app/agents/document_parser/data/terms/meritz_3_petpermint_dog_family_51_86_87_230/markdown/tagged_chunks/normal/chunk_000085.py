from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 갖추어 작성될 것\n'
 '- 4. 전자문서 및 전자서명의 위조ㆍ변조 여부를 확인할 수\n'
 '- 있을 것\n'
 '# 【심신상실자 및 심신박약자】심신상실자(心神喪失者) 또는 심신박약자(心神薄弱者)라\n'
 '함은 정신병, 정신박약, 심한 의식장애 등의 심신장애로\n'
 '인하여 사물 변별 능력 또는 의사 결정 능력이 없거나\n'
 '부족한 자를 말합니다.# 제23조(계약내용의 변경 등)\uf000 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수\n'
 '있습니다. 이 경우 승낙을 서면 등으로 알리거나 보험증권\n'
 '의 뒷면에 기재하여 드립니다.- ① 보험종목\n'
 '- ② 보험기간'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
