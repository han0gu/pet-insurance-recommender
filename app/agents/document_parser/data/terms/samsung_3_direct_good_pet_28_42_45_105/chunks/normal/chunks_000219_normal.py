from langchain_core.documents import Document

chunk = Document(
    page_content=('제20조 (청약의 철회)\n'
 '① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건강상태 진단을 지원하는 계약, '
 '보험기간이 90일 이내인 계약 또는 전문금융 소비자가 체결한 계약은 청약을 철회할 수 없습니다.\n'
 '<용어풀이>\n'
 '[전문금융소비자]\n'
 '보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치단체, 한국은행, 금융회사, '
 '주권상장법인 등을 포함하며 「금융소비자 보호에 관한 법률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 51},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000219',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
